// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// FedTrace : registre d'ancrage de la provenance d'un apprentissage federe.
/// Seules des empreintes (bytes32) et des metadonnees non identifiantes sont stockees :
/// jamais de donnees patient, jamais de poids de modele.
contract FedTrace {
    // ------------------------------------------------------------ structures
    struct Contribution {
        bool exists;
        bytes32 weightsHash;     // empreinte de la mise a jour locale
        bytes32 baseModelHash;   // modele global dont la clinique est partie
        uint32 nSamples;         // nombre d'echantillons declare
        uint64 timestamp;
    }

    struct Model {
        bool exists;
        uint32 round;
        bytes32 parent;          // modele du round precedent
        string strategy;
        uint256 retainedMask;    // bit i = contribution de la clinique i retenue
        uint256 discardedMask;   // bit i = contribution de la clinique i ecartee
        uint32 accuracyBps;      // metriques en points de base (9261 = 0,9261)
        uint32 f1Bps;
        uint64 timestamp;
    }

    struct ModelInput {          // regroupe les parametres de anchorModel
        uint256 round;
        bytes32 modelHash;
        bytes32 parentHash;
        string strategy;
        uint256 retainedMask;
        uint256 discardedMask;
        uint32 accuracyBps;
        uint32 f1Bps;
    }

    struct Decision {
        bytes32 modelHash;
        bytes32 explanationHash;
        uint8 clinicId;
        uint8 predictedClass;
        uint16 probabilityBps;
        uint8 validationStatus;  // 0 = en attente de revue
        uint64 timestamp;
    }

    // ------------------------------------------------------------ etat
    address public owner;
    address public server;
    bool public initialModelAnchored;

    mapping(address => uint8) public clinicIdOf;       // 0 = non enregistree
    mapping(uint8 => address) public clinicAddress;

    mapping(uint256 => mapping(uint8 => Contribution)) private contributions; // round => clinique
    mapping(uint256 => uint8[]) private roundClinics;
    mapping(bytes32 => Model) private models;
    mapping(bytes32 => Decision[]) private decisions;  // historique par cle (pseudonyme du cas)

    // ------------------------------------------------------------ evenements
    event ClinicRegistered(address indexed account, uint8 indexed clinicId);
    event ServerSet(address indexed account);
    event ContributionAnchored(uint256 indexed round, uint8 indexed clinicId, bytes32 weightsHash, bytes32 baseModelHash, uint32 nSamples);
    event ModelAnchored(uint256 indexed round, bytes32 indexed modelHash, bytes32 parentHash, uint256 retainedMask, uint256 discardedMask);
    event DecisionAnchored(bytes32 indexed caseId, bytes32 indexed modelHash, bytes32 explanationHash, uint8 clinicId);

    // ------------------------------------------------------------ droits d'ecriture
    modifier onlyOwner() {
        require(msg.sender == owner, "FedTrace: reserve a l'administrateur");
        _;
    }

    modifier onlyServer() {
        require(msg.sender == server, "FedTrace: reserve au serveur d'agregation");
        _;
    }

    modifier onlyClinic() {
        require(clinicIdOf[msg.sender] != 0, "FedTrace: clinique non autorisee");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // ------------------------------------------------------------ administration
    function registerClinic(address account, uint8 clinicId) external onlyOwner {
        require(account != address(0), "FedTrace: adresse nulle");
        require(clinicId != 0, "FedTrace: identifiant 0 interdit");
        require(clinicIdOf[account] == 0, "FedTrace: adresse deja enregistree");
        require(clinicAddress[clinicId] == address(0), "FedTrace: identifiant deja pris");
        clinicIdOf[account] = clinicId;
        clinicAddress[clinicId] = account;
        emit ClinicRegistered(account, clinicId);
    }

    function setServer(address account) external onlyOwner {
        require(account != address(0), "FedTrace: adresse nulle");
        server = account;
        emit ServerSet(account);
    }

    // ------------------------------------------------------------ ancrage
    /// Racine de l'arbre de provenance : le modele initial (round 0).
    function anchorInitialModel(bytes32 modelHash) external onlyServer {
        require(!initialModelAnchored, "FedTrace: modele initial deja ancre");
        require(modelHash != bytes32(0), "FedTrace: empreinte nulle");
        initialModelAnchored = true;
        models[modelHash] = Model({
            exists: true,
            round: 0,
            parent: bytes32(0),
            strategy: "init",
            retainedMask: 0,
            discardedMask: 0,
            accuracyBps: 0,
            f1Bps: 0,
            timestamp: uint64(block.timestamp)
        });
        emit ModelAnchored(0, modelHash, bytes32(0), 0, 0);
    }

    /// Une clinique ancre l'empreinte de sa mise a jour AVANT l'agregation.
    /// L'identite de la clinique est celle de l'expediteur de la transaction.
    function anchorContribution(
        uint256 round,
        bytes32 weightsHash,
        bytes32 baseModelHash,
        uint32 nSamples
    ) external onlyClinic {
        require(round >= 1, "FedTrace: round invalide");
        require(weightsHash != bytes32(0), "FedTrace: empreinte nulle");
        Model storage base = models[baseModelHash];
        require(base.exists, "FedTrace: modele de depart inconnu");
        require(uint256(base.round) + 1 == round, "FedTrace: le modele de depart n'est pas celui du round precedent");
        uint8 cid = clinicIdOf[msg.sender];
        require(!contributions[round][cid].exists, "FedTrace: contribution deja ancree pour ce round");

        contributions[round][cid] = Contribution({
            exists: true,
            weightsHash: weightsHash,
            baseModelHash: baseModelHash,
            nSamples: nSamples,
            timestamp: uint64(block.timestamp)
        });
        roundClinics[round].push(cid);
        emit ContributionAnchored(round, cid, weightsHash, baseModelHash, nSamples);
    }

    /// Le serveur ancre le modele global. Le contrat REFUSE si une contribution
    /// citee (retenue ou ecartee) n'a pas ete ancree avant : l'ordre est impose par le registre.
    function anchorModel(ModelInput calldata m) external onlyServer {
        require(m.round >= 1, "FedTrace: round invalide");
        require(m.modelHash != bytes32(0), "FedTrace: empreinte nulle");
        require(!models[m.modelHash].exists, "FedTrace: modele deja ancre");
        Model storage parent = models[m.parentHash];
        require(parent.exists, "FedTrace: modele parent inconnu");
        require(uint256(parent.round) + 1 == m.round, "FedTrace: le parent n'est pas le modele du round precedent");
        require(m.retainedMask != 0, "FedTrace: aucune contribution retenue");
        require((m.retainedMask & m.discardedMask) == 0, "FedTrace: contribution a la fois retenue et ecartee");
        _checkContributions(m.round, m.parentHash, m.retainedMask | m.discardedMask);

        models[m.modelHash] = Model({
            exists: true,
            round: uint32(m.round),
            parent: m.parentHash,
            strategy: m.strategy,
            retainedMask: m.retainedMask,
            discardedMask: m.discardedMask,
            accuracyBps: m.accuracyBps,
            f1Bps: m.f1Bps,
            timestamp: uint64(block.timestamp)
        });
        emit ModelAnchored(m.round, m.modelHash, m.parentHash, m.retainedMask, m.discardedMask);
    }

    function _checkContributions(uint256 round, bytes32 parentHash, uint256 all) internal view {
        require((all & 1) == 0, "FedTrace: identifiant de clinique 0 interdit");
        for (uint256 id = 1; id < 256; id++) {
            if (((all >> id) & 1) == 1) {
                Contribution storage c = contributions[round][uint8(id)];
                require(c.exists, "FedTrace: contribution non ancree avant l'agregation");
                require(c.baseModelHash == parentHash, "FedTrace: contribution issue d'un autre modele de depart");
            }
        }
    }

    /// Une clinique ancre un diagnostic. Le contrat refuse tout modele non certifie
    /// (porte de certification appliquee aussi cote registre).
    function anchorDecision(
        bytes32 caseId,
        bytes32 modelHash,
        bytes32 explanationHash,
        uint8 predictedClass,
        uint16 probabilityBps,
        uint8 validationStatus
    ) external onlyClinic {
        require(isCertified(modelHash), "FedTrace: modele non certifie");
        require(predictedClass <= 2, "FedTrace: classe invalide");
        require(probabilityBps <= 10000, "FedTrace: probabilite invalide");
        uint8 cid = clinicIdOf[msg.sender];
        decisions[caseId].push(Decision({
            modelHash: modelHash,
            explanationHash: explanationHash,
            clinicId: cid,
            predictedClass: predictedClass,
            probabilityBps: probabilityBps,
            validationStatus: validationStatus,
            timestamp: uint64(block.timestamp)
        }));
        emit DecisionAnchored(caseId, modelHash, explanationHash, cid);
    }

    // ------------------------------------------------------------ verification et lecture
    /// Certifie = ancre par une federation (round >= 1). Le modele initial n'est pas certifie.
    function isCertified(bytes32 modelHash) public view returns (bool) {
        Model storage m = models[modelHash];
        return m.exists && m.round > 0;
    }

    function verifyModel(bytes32 modelHash)
        external
        view
        returns (bool certified, uint256 round, uint256 retainedMask, uint256 discardedMask, uint32 accuracyBps)
    {
        Model storage m = models[modelHash];
        return (m.exists && m.round > 0, m.round, m.retainedMask, m.discardedMask, m.accuracyBps);
    }

    function verifyDecision(bytes32 caseId, bytes32 explanationHash) external view returns (bool) {
        Decision[] storage list = decisions[caseId];
        for (uint256 i = 0; i < list.length; i++) {
            if (list[i].explanationHash == explanationHash) {
                return true;
            }
        }
        return false;
    }

    function getModel(bytes32 modelHash) external view returns (Model memory) {
        require(models[modelHash].exists, "FedTrace: modele inconnu");
        return models[modelHash];
    }

    /// Provenance : du modele jusqu'au modele initial (hashes[0] = le modele demande).
    function getProvenance(bytes32 modelHash) external view returns (bytes32[] memory hashes, uint256[] memory rounds) {
        require(models[modelHash].exists, "FedTrace: modele inconnu");
        uint256 n = uint256(models[modelHash].round) + 1;
        hashes = new bytes32[](n);
        rounds = new uint256[](n);
        bytes32 h = modelHash;
        for (uint256 i = 0; i < n; i++) {
            hashes[i] = h;
            rounds[i] = models[h].round;
            h = models[h].parent;
        }
    }

    function getRoundContributions(uint256 round)
        external
        view
        returns (uint8[] memory ids, bytes32[] memory weightsHashes, bytes32[] memory baseHashes, uint32[] memory nSamples)
    {
        uint8[] storage list = roundClinics[round];
        uint256 n = list.length;
        ids = new uint8[](n);
        weightsHashes = new bytes32[](n);
        baseHashes = new bytes32[](n);
        nSamples = new uint32[](n);
        for (uint256 i = 0; i < n; i++) {
            Contribution storage c = contributions[round][list[i]];
            ids[i] = list[i];
            weightsHashes[i] = c.weightsHash;
            baseHashes[i] = c.baseModelHash;
            nSamples[i] = c.nSamples;
        }
    }

    /// Historique complet d'une cle (un cas).
    function getDecisions(bytes32 caseId) external view returns (Decision[] memory) {
        return decisions[caseId];
    }
}
